; Domain Description
; 
; This domain is for building towers using Pick and Place actions.
; Blocks may start anywhere on the table or in configurations that 
; have towers themselves.
;
; There are two types of objects in this world: Blocks and the Table.
; Blocks can be moved but the Table is fixed.

(define (domain tool-use)
  (:requirements :strips :equality)
  (:predicates
    ; Types
    (Block ?b)
    (Table ?o)
    (Pose ?o ?p) 
    (RelPose ?o1 ?o2 ?rp)   
    (Grasp ?o ?g)
    (Conf ?q)
    (StartConf ?q)
    (EndConf ?q)

    ; Kin: Object ?o at Pose ?p can be grasped at Grasp ?g achievable with config ?q and traj ?t
    ; The trajectories differ for pick and place actions.
    (PlaceKin ?o ?p ?g ?q1 ?q2 ?t)           
    (PickKin ?o ?p ?g ?q1 ?q2 ?t)          
    ; FreeMotion: ?t is a traj between configurations ?q1 and ?q2
    (FreeMotion ?q1 ?t ?q2)
    ; HoldingMotion: ?t is a traj between ?q1 and ?q2 while holding object ?o with grasp ?g 
    (HoldingMotion ?q1 ?t ?q2 ?o ?g)
    ; Supported: Block ?o at Pose ?p1 will be supported by Table or Object ?o2 at Pose ?p2.
    (Supported ?o1 ?p1 ?o2 ?p2)
    ; Home: Block ?o at Pose ?p1 will be at its home position on Object ?o2 at Pose ?p2.
    (Home ?o1 ?p1 ?o2 ?p2)

    ; Fluents 
    (On ?o1 ?o2)
    (AtPose ?o ?p)
    (AtGrasp ?o ?g)
    (HandEmpty)
    (AtConf ?q)
    (AtHome ?o)
    (CanMove)
    (Reset)

    ; Derived
    ; Stackable: Blocks can only have one object placed on them. A Table is always Stackable
    ;            and a Block is Stackable if there is nothing on it.
    (Stackable ?o)
    ; Movable: Blocks that have nothing on top of them are Movable.
    (Movable ?o)
  )

  (:action move_free
    :parameters (?q1 ?q2 ?t)
    :precondition (and (FreeMotion ?q1 ?t ?q2)
                       (AtConf ?q1) 
                       (HandEmpty) 
                       (CanMove)
                       (StartConf ?q1)
                       (EndConf ?q2))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)) 
                 (not (CanMove)))
  )

  (:action move_holding
    :parameters (?q1 ?q2 ?o ?g ?t)
    :precondition (and (HoldingMotion ?q1 ?t ?q2 ?o ?g)
                       (AtConf ?q1) 
                       (AtGrasp ?o ?g) 
                       (CanMove)
                       (StartConf ?q1)
                       (EndConf ?q2))
    :effect (and (AtConf ?q2)
                 (not (AtConf ?q1)) 
                 (not (CanMove)))
  )
    
  ; Pick up Block ?o1 at Pose ?p1 from Object (Block or Table) ?o2
  (:action pick
    :parameters (?o1 ?p1 ?o2 ?g ?q1 ?q2 ?t)
    :precondition (and (PickKin ?o1 ?p1 ?g ?q1 ?q2 ?t) 
                       (AtPose ?o1 ?p1) 
                       (HandEmpty) 
                       (AtConf ?q1)
                       (Movable ?o1) 
                       (not (CanMove)) 
                       (On ?o1 ?o2)
                       (EndConf ?q1)
                       (StartConf ?q2))
    :effect (and (AtGrasp ?o1 ?g) 
                 (CanMove)
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtPose ?o1 ?p1)) 
                 (not (HandEmpty))
                 (not (On ?o1 ?o2))
                 (not (AtHome ?o1)))
  )
  
  ; Place Block ?o at Pose ?p1 on Object (Block or Table) ?o2 which is at Pose ?p2
  (:action place
    :parameters (?o1 ?p1 ?o2 ?p2 ?g ?q1 ?q2 ?t)
    :precondition (and (PlaceKin ?o1 ?p1 ?g ?q1 ?q2 ?t)
                       (AtGrasp ?o1 ?g) 
                       (AtConf ?q1) 
                       (Supported ?o1 ?p1 ?o2 ?p2)
                       (AtPose ?o2 ?p2) 
                       (Stackable ?o2)
                       (EndConf ?q1)
                       (StartConf ?q2) 
                       (not (CanMove)))
    :effect (and (AtPose ?o1 ?p1) 
                 (HandEmpty) 
                 (CanMove) 
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtGrasp ?o1 ?g)) 
                 (On ?o1 ?o2))
  )

  ; Place Block ?o at Pose ?p1 on Object (Block or Table) ?o2 which is at Pose ?p2
  (:action place_home
    :parameters (?o1 ?p1 ?o2 ?p2 ?g ?q1 ?q2 ?t)
    :precondition (and (Reset)
                       (PlaceKin ?o1 ?p1 ?g ?q1 ?q2 ?t)
                       (AtGrasp ?o1 ?g) 
                       (AtConf ?q1) 
                       (Home ?o1 ?p1 ?o2 ?p2)
                       (AtPose ?o2 ?p2) 
                       (Stackable ?o2)
                       (EndConf ?q1)
                       (StartConf ?q2) 
                       (not (CanMove)))
    :effect (and (AtHome ?o1)
                 (AtPose ?o1 ?p1) 
                 (HandEmpty) 
                 (CanMove) 
                 (AtConf ?q2)
                 (not (AtConf ?q1))
                 (not (AtGrasp ?o1 ?g)) 
                 (On ?o1 ?o2))
  )

  (:derived (Stackable ?o1)
    (or (forall (?o2) (not (On ?o2 ?o1))) (Table ?o1))
  )

  (:derived (Movable ?o)
    (and (Block ?o) (not (exists (?o2) (On ?o2 ?o))))
  )

)
